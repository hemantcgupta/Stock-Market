import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi';
import { images } from '../../Constants/images';
import DataTableComponent from '../../Component/DataTableComponent';
import TotalRow from '../../Component/TotalRow';
import RegionsDropDown from '../../Component/RegionsDropDown';
import Pagination from '../../Component/pagination';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Switch from '@material-ui/core/Switch';
import color from '../../Constants/color'
import $ from 'jquery';
import '../../App';
import './TopMarkets.scss';
import TopMenuBar from '../../Component/TopMenuBar';
import _ from 'lodash';
import DownloadCustomHeaderGroup from './DownloadCustomHeaderGroup';

const apiServices = new APIServices();

class TopMarkets extends Component {
  constructor(props) {
    super(props);
    this.state = {
      startDate: null,
      endDate: null,
      regionSelected: 'Null',
      countrySelected: 'Null',
      citySelected: 'Null',
      selectedOD: '',
      ODSelected: '',
      topMarketColumn: [],
      topMarketData: [],
      topMarketTotal: [],
      topCompetitorColumn: [],
      topAgentColumn: [],
      topCompetitorData: [],
      topCompetitorTotal: [],
      topAgentData: [],
      topAgentTotal: [],
      odName: '',
      competitorName: '',
      agentName: '',
      modelRegionDatas: [],
      modelregioncolumn: [],
      tableDatas: true,
      tableTitle1: '',
      tableTitle2: '',
      // monthTableTitle:'',
      top5ODTitle: 'TOP 5 ODs',
      tabLevel: 'Null',
      posFlowData: 'Null',
      posAgentFlowDatas: [],
      posAgentFlowcolumn: [],
      getCabinValue: [],
      toggleChange: false,
      loading: false,
      loading2: false,
      loading3: false,
      topMarketRowHighlighted: {
        'highlight-row': 'data.highlightMe',
      },
      competitorRowHighlighted: {
        'highlight-row': 'data.highlightMe',
      },
      agentRowHighlighted: {
        'highlight-row': 'data.highlightMe',
      },
      currentPage: '',
      totalPages: '',
      totalRecords: '',
      paginationStart: 1,
      paginationEnd: '',
      paginationSize: '',
      count: 1,
      searchClicked: false
    };
    this.sendEvent('1', 'viewed Top Markets Page', 'topMarkets', 'Top Markets');
  }

  sendEvent = (id, description, path, page_name) => {
    var eventData = {
      event_id: `${id}`,
      description: `User ${description}`,
      where_path: `/${path}`,
      page_name: `${page_name} Page`
    }
    eventApi.sendEvent(eventData)
  }

  componentWillMount() {
    const self = this;
    apiServices.getClassNameDetails().then((result) => {
      if (result) {
        var classData = result[0].classDatas;
        self.setState({ cabinOption: classData })
      }
    });
    // let OD = window.localStorage.getItem('ODSelected')
    // if (OD !== null && OD !== 'Null') {
    //   self.setState({ ODSelected: OD })
    // }
  }

  getFilterValues = ($event) => {
    this.setState({
      regionSelected: $event.regionSelected === 'All' ? 'Null' : $event.regionSelected,
      countrySelected: $event.countrySelected,
      citySelected: $event.citySelected,
      ODSelected: $event.ODSelected,
      endDate: $event.endDate,
      startDate: $event.startDate,
      getCabinValue: $event.getCabinValue,
    }, () => this.gotoFirstPage())
  }

  getTopMarketData = () => {
    var self = this;
    let { startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, ODSelected, selectedOD, count } = this.state;
    self.setState({ loading: true, loading2: true, loading3: true, topMarketData: [], topMarketTotal: [] })

    apiServices.getTopMarkets(count, startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, ODSelected).then(function (result) {
      self.setState({ loading: false })
      if (result) {
          var rowData = result[0].rowData;
          rowData = rowData.map((data, index) => {
            if (selectedOD) {
              data.highlightMe = selectedOD === data.OD
            } else if (index === 0) {
              data.highlightMe = true;
            }
            return data;
          })
          if (self.state.searchClicked) {
            if (rowData.length < result[0].paginationSize) {
              self.setState({ paginationEnd: result[0].totalRecords }, () => self.setState({ searchClicked: false }))
            } else {
              self.setState({ paginationEnd: result[0].paginationSize }, () => self.setState({ searchClicked: false }))
            }
          }
          self.setState({
            topMarketData: rowData,
            topMarketTotal: result[0].totalData,
            topMarketColumn: result[0].columnName,
            currentPage: result[0].currentPage,
            totalPages: result[0].totalPages,
            totalRecords: result[0].totalRecords,
            paginationSize: result[0].paginationSize,
            odName: selectedOD ? selectedOD : '',
            tableTitle1: selectedOD ? `Top Competitors for ${selectedOD}` : ''
          }, () => self.getCompetitorsData())
        }
    });
  }

  getCompetitorsData = () => {
    var self = this;
    self.setState({ loading2: true, loading3: true })
    let { startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, topMarketData, odName } = this.state;
    if (!odName) {
      var odNameFirst = topMarketData.length > 0 ? topMarketData[0].OD : 'Null';
      self.setState({
        tableTitle1: `Top Competitors for ${odNameFirst}`,
        odName: odNameFirst ? odNameFirst : 'Null',
      })
    }
    apiServices.getTopCompetitors(startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, !odName ? odNameFirst : odName, 'Null').then(function (result) {
      self.setState({ loading2: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;

        rowData = rowData.map((data, index) => {
          if (index === 0) {
            data.highlightMe = true;
          }
          return data;
        })

        self.setState({ topCompetitorData: rowData, topCompetitorTotal: totalData, topCompetitorColumn: columnName }, () => self.getAgentsData())
      }
    });
  }

  getAgentsData = () => {
    var self = this;
    self.setState({ loading3: true })
    let { startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, topCompetitorData, odName } = this.state;
    var competitorName = topCompetitorData.length > 0 ? topCompetitorData[0].Airline : 'Null'
    self.setState({
      tableTitle2: `Top Agents for ${odName} & ${competitorName}`,
      competitorName: competitorName
    })
    apiServices.getTopAgents(startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, odName, competitorName).then(function (result) {
      self.setState({ loading3: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        self.setState({ topAgentData: rowData, topAgentTotal: totalData, topAgentColumn: columnName })
      }
    });
  }

  getAgentsDataOnClick = () => {
    var self = this;
    self.setState({ loading3: true })
    let { startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, odName, competitorName } = this.state;
    apiServices.getTopAgents(startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, odName, competitorName).then(function (result) {
      self.setState({ loading3: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        self.setState({ topAgentData: rowData, topAgentTotal: totalData, topAgentColumn: columnName })
      }
    });
  }

  getSwappedTopMarketData = () => {
    var self = this;
    let { startDate, endDate, regionSelected, countrySelected, citySelected, ODSelected, getCabinValue, count } = this.state;
    self.setState({ loading: true, loading2: true, loading3: true, topMarketData: [], topMarketTotal: [] })
    apiServices.getTopMarkets(count, startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, ODSelected).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var rowData = result[0].rowData;
        rowData = rowData.map((data, index) => {
          if (index === 0) {
            data.highlightMe = true;
          }
          return data;
        })
        if (self.state.searchClicked) {
          if (rowData.length < result[0].paginationSize) {
            self.setState({ paginationEnd: result[0].totalRecords }, () => self.setState({ searchClicked: false }))
          } else {
            self.setState({ paginationEnd: result[0].paginationSize }, () => self.setState({ searchClicked: false }))
          }
        }
        self.setState({
          topMarketData: rowData,
          topMarketTotal: result[0].totalData,
          topMarketColumn: result[0].columnName,
          currentPage: result[0].currentPage,
          totalPages: result[0].totalPages,
          totalRecords: result[0].totalRecords,
          paginationSize: result[0].paginationSize
        }, () => self.getSwappedAgentsData())
      }
    });
  }

  getSwappedAgentsData = () => {
    var self = this;
    self.setState({ loading2: true, loading3: true })
    let { startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, odName } = this.state;
    apiServices.getTopAgents(startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, odName, 'Null').then(function (result) {
      self.setState({ loading3: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;

        rowData = rowData.map((data, index) => {
          if (index === 0) {
            data.highlightMe = true;
          }
          return data;
        })

        self.setState({ topAgentData: rowData, topAgentTotal: totalData, topAgentColumn: columnName }, () => self.getSwappedCompetitorsData())
      }
    });
  }

  getSwappedCompetitorsData = () => {
    var self = this;
    self.setState({ loading2: true })
    let { startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, topAgentData, odName } = this.state;
    var agentName = topAgentData.length > 0 ? topAgentData[0].Name : ''
    self.setState({
      tableTitle1: `Top Competitors for ${odName} & ${agentName}`,
      agentName: agentName
    })
    apiServices.getTopCompetitors(startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, odName, agentName).then(function (result) {
      self.setState({ loading2: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        self.setState({ topCompetitorData: rowData, topCompetitorTotal: totalData, topCompetitorColumn: columnName })
      }
    });
  }

  getSwappedCompetitorsDataOnClick = () => {
    var self = this;
    self.setState({ loading2: true })
    let { startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, odName, agentName } = this.state;
    apiServices.getTopCompetitors(startDate, endDate, regionSelected, countrySelected, citySelected, getCabinValue, odName, agentName).then(function (result) {
      self.setState({ loading2: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        self.setState({ topCompetitorData: rowData, topCompetitorTotal: totalData, topCompetitorColumn: columnName })
      }
    });
  }

  odRowClick = (params) => {
    this.sendEvent('2', 'clicked on OD', 'topMarkets', 'Top Markets');
    var column = params.colDef.field;
    var rowData = params.api.getSelectedRows()
    var odName = rowData.length > 0 ? rowData[0].OD : '';
    var competitorName = this.state.topCompetitorData.length > 0 ? this.state.topCompetitorData[0].Airline : ''
    var agentName = this.state.topAgentData.length > 0 ? this.state.topAgentData[0].Name : ''

    const topMarketData = this.state.topMarketData.map((d) => {
      d.highlightMe = false;
      return d;
    })
    params.api.updateRowData({ update: topMarketData });

    if (column === 'OD') {
      if (this.state.toggleChange) {
        this.setState({
          tableTitle1: `Top Competitors for ${odName} & ${agentName}`,
          tableTitle2: `Top Agents for ${odName}`,
          odName: odName,
          competitorName: competitorName,
          agentName: agentName,
        }, () => this.getSwappedAgentsData())
      } else {
        this.setState(
          {
            tableTitle1: `Top Competitors for ${odName}`,
            tableTitle2: `Top Agents for ${odName} & ${competitorName}`,
            odName: odName,
            competitorName: competitorName,
          },
          () => this.getCompetitorsData()
        );
      }
    }
    window.localStorage.setItem('ODSelected', 'Null')
    this.setState({ selectedOD: '' })
  }

  competitorCellClick = (params) => {
    this.sendEvent('2', 'clicked on Competitor', 'topMarkets', 'Top Markets');
    var column = params.colDef.field;
    const topCompetitorData = this.state.topCompetitorData.map((d) => {
      d.highlightMe = false;
      return d;
    })
    params.api.updateRowData({ update: topCompetitorData });
    if (column === 'Airline') {
      window.localStorage.setItem('Competitor', params.value)
      this.props.history.push('/competitorAnalysis')
    } else if (column === 'CY_MIDTA') {
      if (!this.state.toggleChange) {
        var competitorName = params.data.Airline;
        this.setState({
          competitorName: competitorName,
          tableTitle2: `Top Agents for ${this.state.odName} & ${competitorName}`,
        }, () => this.getAgentsDataOnClick())
      }
    }
  }

  agentCellClick = (params) => {
    this.sendEvent('2', 'clicked on Agent', 'topMarkets', 'Top Markets');
    var column = params.colDef.field;
    var agentName = params.data.Name

    const topAgentData = this.state.topAgentData.map((d) => {
      d.highlightMe = false;
      return d;
    })
    params.api.updateRowData({ update: topAgentData });

    if (column === 'Name') {
      window.localStorage.setItem('Agent', params.value)
      this.props.history.push('/agentAnalysis')
    } else if (column === 'CY_MIDT') {
      if (this.state.toggleChange) {
        this.setState({
          agentName: agentName,
          tableTitle1: `Top Competitors for ${this.state.odName} & ${agentName}`,
        }, () => this.getSwappedCompetitorsDataOnClick())
      }
    }
  }

  toggleChecked = () => {
    this.sendEvent('2', 'clicked on Swap button', 'topMarkets', 'Top Markets');
    this.setState({ toggleChange: !this.state.toggleChange }, () => this.toggleChange())
  }

  toggleChange = () => {
    var agentName = this.state.topAgentData[0].Name
    var competitorName = this.state.topCompetitorData[0].Airline
    if (this.state.toggleChange) {
      this.setState({
        tableTitle1: `Top Competitors for ${this.state.odName}`,
        tableTitle2: `Top Agents for ${this.state.odName}`,
        agentName: agentName,
        competitorName: competitorName
      }, () => this.getSwappedAgentsData())
    } else {
      this.setState({
        tableTitle1: `Top Competitors for ${this.state.odName} `,
        tableTitle2: `Top Agents for ${this.state.odName} & ${competitorName}`,
        agentName: agentName,
        competitorName: competitorName
      }, () => this.getCompetitorsData())
    }

  }

  search = () => {
    this.sendEvent('2', 'clicked on Search button', 'topMarkets', 'Top Markets');
    if (this.state.toggleChange) {
      this.setState({ searchClicked: true }, () => this.getSwappedTopMarketData())
    } else {
      this.setState({ odName: '', searchClicked: true }, () => this.getTopMarketData())
    }
  }

  paginationClick = () => {
    if (this.state.toggleChange) {
      this.getSwappedTopMarketData();
    } else {
      this.setState({ odName: '' }, () => this.getTopMarketData())
    }
  }

  renderTopSection = () => {

    return (
      <div className="navdesign">
        <div className="col-md-12 col-sm-12 col-xs-12">
          <section>
            <h2>Top Markets</h2>
          </section>
        </div>

        <RegionsDropDown
          pageName={'top_markets'}
          getFilterValues={this.getFilterValues}
          {...this.props} />

      </div>
    )
  }

  gotoFirstPage = () => {
    this.setState({
      count: 1,
      paginationStart: 1,
    },
      () => {
        this.search();
      })
  }

  gotoLastPage = () => {
    const { totalPages, paginationSize, totalRecords } = this.state;
    const startDigit = paginationSize * (totalPages - 1)
    this.setState({
      count: totalPages,
      paginationStart: startDigit + 1,
      paginationEnd: totalRecords
    },
      () => this.paginationClick())
  }

  gotoPreviousPage = () => {
    const { count, currentPage, totalPages, paginationSize, paginationStart, paginationEnd, totalRecords } = this.state;
    const remainder = totalRecords % paginationSize
    const fromLast = currentPage === totalPages
    this.setState({
      count: count - 1,
      paginationStart: paginationStart - paginationSize,
      paginationEnd: paginationEnd - (fromLast ? remainder : paginationSize)
    },
      () => this.paginationClick())
  }

  gotoNextPage = () => {
    const { count, currentPage, totalPages, paginationSize, paginationStart, paginationEnd, totalRecords } = this.state;
    const remainder = totalRecords % paginationSize
    const tolast = currentPage === totalPages - 1
    this.setState({
      count: count + 1,
      paginationStart: paginationStart + paginationSize,
      paginationEnd: paginationEnd + (tolast ? remainder : paginationSize)
    },
      () => this.paginationClick())
  }

  render() {
    const { loading, loading2, loading3, topMarketData, topMarketTotal, topMarketColumn, topCompetitorData, topCompetitorTotal, topCompetitorColumn, topAgentData, topAgentTotal, topAgentColumn, tableTitle1, tableTitle2, toggleChange, cabinOption } = this.state;
    return (
      <div className='top-market'>
        <TopMenuBar {...this.props} swap={toggleChange} />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 top-market-main">

            {this.renderTopSection()}

            <div className="x_panel" style={{ marginTop: "10px" }}>
              <div className="x_content">

                <DataTableComponent
                  rowData={topMarketData}
                  columnDefs={topMarketColumn}
                  onCellClicked={(cellData) => this.odRowClick(cellData)}
                  frameworkComponents={{ customHeaderGroupComponent: DownloadCustomHeaderGroup }}
                  topMarket={true}
                  loading={loading}
                  rowClassRules={this.state.topMarketRowHighlighted}
                />
                <TotalRow
                  rowData={topMarketTotal}
                  columnDefs={topMarketColumn}
                  frameworkComponents={{ customHeaderGroupComponent: DownloadCustomHeaderGroup }}
                  loading={loading}
                />
                {loading ? '' :
                  <Pagination
                    paginationStart={this.state.paginationStart}
                    paginationEnd={this.state.paginationEnd}
                    totalRecords={this.state.totalRecords}
                    currentPage={this.state.currentPage}
                    TotalPages={this.state.totalPages}
                    gotoFirstPage={() => this.gotoFirstPage()}
                    gotoLastPage={() => this.gotoLastPage()}
                    gotoPreviousPage={() => this.gotoPreviousPage()}
                    gotoNexttPage={() => this.gotoNextPage()}
                  />}

                <div className="tab" id="posTableTab" role="tabpanel" style={{ padding: '0px' }}>
                  <div className="tab-content tabs">
                    <div role="tabpanel" className="tab-pane fade in active" id="Section1">

                      {toggleChange ?
                        <div className='row' style={{ display: 'flex' }}>
                          <div className='col-md-6 col-sm-6 col-xs-12'>
                            <div className='toggle-agent-competitor' style={{ display: topAgentData.length > 0 ? 'flex' : 'block' }}>
                              {loading3 ? '' : topAgentData.length > 0 ? <h2 className='table-name'>{tableTitle2}</h2> : <h2 />}
                            </div>
                            <DataTableComponent
                              rowData={topAgentData}
                              columnDefs={topAgentColumn}
                              onCellClicked={(cellData) => this.agentCellClick(cellData)}
                              topMarket={true}
                              loading={loading3}
                              rowClassRules={this.state.agentRowHighlighted}
                            />
                            <TotalRow
                              rowData={topAgentTotal}
                              columnDefs={topAgentColumn}
                              loading={loading3}
                            />
                          </div>
                          {loading3 ? '' : topAgentData.length > 0 ? <span className='swap' onClick={() => this.toggleChecked()}><i class={`fa fa-exchange transition rotate`} aria-hidden="true"></i></span> : ''}
                          <div className='col-md-6 col-sm-6 col-xs-12'>
                            {loading2 ? '' : topCompetitorData.length > 0 ? <h2 className='table-name'>{tableTitle1}</h2> : <h2 />}
                            <DataTableComponent
                              rowData={topCompetitorData}
                              columnDefs={topCompetitorColumn}
                              onCellClicked={(cellData) => this.competitorCellClick(cellData)}
                              topMarket={true}
                              loading={loading2}
                            />
                            <TotalRow
                              rowData={topCompetitorTotal}
                              columnDefs={topCompetitorColumn}
                              loading={loading2}
                            />
                          </div>
                        </div>
                        :
                        <div className='row' style={{ display: 'flex' }}>
                          <div className='col-md-6 col-sm-6 col-xs-12'>
                            <div className='toggle-agent-competitor' style={{ display: topCompetitorData.length > 0 ? 'flex' : 'block' }}>
                              {loading2 ? '' : topCompetitorData.length > 0 ? <h2 className='table-name'>{tableTitle1}</h2> : <h2 />}
                              {/* <li className='switch'>
                                <FormControlLabel
                                  control={<Switch checked={toggleChange} onChange={() => this.toggleChecked()} />}
                                  label=""
                                />
                              </li> */}
                            </div>
                            <DataTableComponent
                              rowData={topCompetitorData}
                              columnDefs={topCompetitorColumn}
                              onCellClicked={(cellData) => this.competitorCellClick(cellData)}
                              topMarket={true}
                              loading={loading2}
                              rowClassRules={this.state.competitorRowHighlighted}
                            />
                            <TotalRow
                              rowData={topCompetitorTotal}
                              columnDefs={topCompetitorColumn}
                              loading={loading2}
                            />
                          </div>
                          {loading2 ? '' : topCompetitorData.length > 0 ? <span className='swap' onClick={() => this.toggleChecked()}><i class={`fa fa-exchange transition rotate`} aria-hidden="true"></i></span> : ''}
                          <div className='col-md-6 col-sm-6 col-xs-12'>
                            {loading3 ? '' : topAgentData.length > 0 ? <h2 className='table-name'>{tableTitle2}</h2> : <h2 />}
                            <DataTableComponent
                              rowData={topAgentData}
                              columnDefs={topAgentColumn}
                              onCellClicked={(cellData) => this.agentCellClick(cellData)}
                              topMarket={true}
                              loading={loading3}
                            />
                            <TotalRow
                              rowData={topAgentTotal}
                              columnDefs={topAgentColumn}
                              loading={loading3}
                            />
                          </div>
                        </div>}
                    </div>

                  </div>
                </div>

              </div>
            </div>
          </div>
        </div>
      </div>

    );
  }
}

export default TopMarkets;