import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import api from '../../API/api';
import eventApi from '../../API/eventApi';
import DataTableComponent from '../../Component/DataTableComponent';
import RegionsDropDown from '../../Component/RegionsDropDown';
import TotalRow from '../../Component/TotalRow';
import '../../App';
import './CompetitorAnalysis.scss';
import TopMenuBar from '../../Component/TopMenuBar';
import AutoSelectPagination from '../../Component/AutoSelectPagination';
import _ from 'lodash'

const apiServices = new APIServices();

class CompetitorAnalysis extends Component {
  constructor(props) {
    super(props);
    this.state = {
      startDate: null,
      endDate: null,
      regionSelected: 'Null',
      countrySelected: 'Null',
      citySelected: 'Null',
      airline: [],
      competitorName: '',
      competitorValue: [],
      topMarketNum: ['5', '10', '15', '20'],
      selectedTopMarketNum: '5',
      competitorColumn: [],
      competitorData: [],
      total: [],
      topMarketColumn: [],
      topMarketData: [],
      topMarketTotal: [],
      topAgentColumn: [],
      topAgentData: [],
      topAgentTotal: [],
      odName: '',
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
      odName: '',
      loading: false,
      loading2: false,
      loading3: false,
      topMarketRowHighlighted: {
        'highlight-row': 'data.highlightMe',
      },
    };
    this.sendEvent('1', 'viewed Competitor Analysis Page', 'competitorAnalysis', 'Competitor Analysis');
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

  componentDidMount() {
    // var self = this;
    var competitorName = window.localStorage.getItem('Competitor')
    this.setState({
      competitorName: competitorName !== null ? competitorName : '',
      competitorValue: [{ value: '1', label: competitorName }],
      tableTitle1: `Top Markets for ${competitorName}`,
    })

    apiServices.getClassNameDetails().then((result) => {
      if (result) {
        var classData = result[0].classDatas;
        this.setState({ cabinOption: classData })
      }
    });

  }

  getFilterValues = ($event) => {
    this.setState({
      regionSelected: $event.regionSelected === 'All' ? 'Null' : $event.regionSelected,
      countrySelected: $event.countrySelected,
      citySelected: $event.citySelected,
      startDate: $event.startDate,
      endDate: $event.endDate,
      getCabinValue: $event.getCabinValue,
    }, () => { this.getCompetitorAnalysis(); this.getTopMarketData(); })
  }

  getCompetitorAnalysis = () => {
    var self = this;
    let { endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, competitorName } = this.state;
    self.setState({ loading: true })
    let twoCharCompetitorName = competitorName.substring(0, 2)
    apiServices.getCompetitorAnalysis(endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, twoCharCompetitorName).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        self.setState({ competitorColumn: result[0].columnName, competitorData: result[0].rowData, total: result[0].totalData })
      }
    });
  }

  getTopMarketData = () => {
    var self = this;
    let { endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, competitorName, selectedTopMarketNum } = this.state;
    self.setState({ loading2: true, loading3: true })
    let twoCharCompetitorName = competitorName.substring(0, 2)
    apiServices.getTopMarketsForCompetitors(endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, twoCharCompetitorName, selectedTopMarketNum).then(function (result) {
      self.setState({ loading2: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        var odName = rowData.length > 0 ? rowData[0].OD : '';

        rowData = rowData.map((data, index) => {
          if (index === 0) {
            data.highlightMe = true;
          }
          return data;
        })

        self.setState({
          topMarketData: rowData,
          topMarketTotal: totalData,
          topMarketColumn: columnName,
          odName: odName,
          tableTitle1: `Top Markets for ${competitorName}`,
          tableTitle2: `Top 5 Agents for ${competitorName} & ${odName}`,
        },
          () => self.getAgentsData())
      }
    });
  }

  getAgentsData = () => {
    var self = this;
    self.setState({ loading3: true })
    let { endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, competitorName, selectedTopMarketNum, odName } = self.state;
    let twoCharCompetitorName = competitorName.substring(0, 2)
    apiServices.getTopAgentsForCompetitors(endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, twoCharCompetitorName, selectedTopMarketNum, odName).then(function (result) {
      self.setState({ loading3: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        self.setState({ topAgentData: rowData, topAgentTotal: totalData, topAgentColumn: columnName })
      }
    });
  }

  callAirline = (value) => {
    this.sendEvent('2', 'clicked on Airlines Dropdown', 'competitorAnalysis', 'Competitor Analysis');
    this.setState({ competitorName: value.label, competitorValue: value }, () => { this.getCompetitorAnalysis(); this.getTopMarketData(); })
  }

  callTopMarket = (e) => {
    this.sendEvent('2', 'clicked on Top Markets Dropdown', 'competitorAnalysis', 'Competitor Analysis');
    this.setState({ loading: false, selectedTopMarketNum: e.target.value }, () => this.getTopMarketData())
  }

  topMarketRowClick = (params) => {
    this.sendEvent('2', 'clicked on OD', 'competitorAnalysis', 'Competitor Analysis');
    var rowData = params.api.getSelectedRows();
    var odName = rowData[0].OD;
    var column = params.colDef.field;

    const topMarketData = this.state.topMarketData.map((d) => {
      d.highlightMe = false;
      return d;
    })
    params.api.updateRowData({ update: topMarketData });

    if (column === 'OD') {
      this.setState({
        topMarketRowHighlighted: null,
        odName: odName,
        tableTitle2: `Top 5 Agents for ${this.state.competitorName} & ${odName}`
      }, () => this.getAgentsData())
    }
  }

  agentCellClick = (params) => {
    this.sendEvent('2', 'clicked on Agent', 'competitorAnalysis', 'Competitor Analysis');
    var column = params.colDef.field;
    if (column === 'Name') {
      window.localStorage.setItem('Agent', params.value)
      // window.localStorage.setItem('Month', this.state.gettingMonth)
      this.props.history.push('/agentAnalysis')
    }
  }

  search = () => {
    this.sendEvent('2', 'clicked on Search button', 'competitorAnalysis', 'Competitor Analysis');
    this.getCompetitorAnalysis();
    this.getTopMarketData();
  }

  serialize = (params) => {
    var str = [];
    for (var p in params)
      if (params.hasOwnProperty(p)) {
        str.push(encodeURIComponent(p) + "=" + encodeURIComponent(params[p]));
      }
    return str.join("&");
  }

  getAirlines = (airline_name, pageNumber) => {
    return new Promise((resolve, reject) => {
      const params = {};
      if (airline_name) {
        params['competitor_name'] = airline_name
      }
      if (pageNumber) {
        params['page_number'] = pageNumber
      }
      api.get(`getcompetitor?${this.serialize(params)}`)
        .then((response) => {
          if (response) {
            if (response.data.response) {
              let data = response.data.response;
              resolve({
                airlines: data.events,
                totalRecords: data.totalAgents,
                paginationSize: data.paginationLimit,
                totalPages: data.totalPages
              })
            }
          }
        })
        .catch((err) => {
          resolve(null)
        })
    })

  }

  loadOptions = async (search, page) => {

    const result = await this.getAirlines(search, page);
    if (result === null) {
      return
    }
    const {
      airlines,
      totalRecords,
      totalPages
    } = result;

    if (airlines.length > 0) {
      airlines.map(airline => {
        if (airline.label) {
          const s = airline.label.trim().toLowerCase();
          airline.label = s.charAt(0).toUpperCase() + s.slice(1);
        }
        return airline
      });
      return {
        options: airlines,
        hasMore: totalPages > page + 1
      };
    }
  };

  render() {
    const { loading, loading2, loading3, topMarketData, topMarketColumn, competitorData, topMarketTotal, competitorColumn, total, topAgentData, topAgentTotal, topAgentColumn, tableTitle1, tableTitle2, topMarketNum } = this.state;
    const topMarketOptionItems = topMarketNum.map((num) =>
      <option value={num}>{num}</option>
    );

    return (
      <div className='competitor-analysis'>
        <TopMenuBar {...this.props} />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 competitor-analysis-main">
            <div className="navdesign">
              <div className="col-md-12 col-sm-12 col-xs-12 top-heading">
                <section>
                  <h2>Competitor Analysis</h2>
                </section>
                <div style={{ display: 'flex' }}>
                  <div className="" >
                    <div className="form-group" style={{ maxWidth: '300px' }}>
                      <h4>Airline :</h4>
                      <AutoSelectPagination
                        value={this.state.competitorValue}
                        onChange={(value) => this.callAirline(value)}
                        loadOptions={this.loadOptions}
                      />
                    </div>
                  </div>
                  <div className="" >
                    <div className="form-group" style={{ maxWidth: '155px' }}>
                      <h4 style={{ width: '280px' }}>Top Markets:</h4>
                      <select className="form-control cabinselect" onChange={(e) => this.callTopMarket(e)}>
                        {topMarketOptionItems}
                      </select>
                    </div>
                  </div>
                </div>
              </div>

              <div className="drop-down-main">

                <RegionsDropDown
                  getFilterValues={this.getFilterValues}
                  {...this.props} />

                {/* <div className="">
                  <button type="button" className="btn search" onClick={() => this.search()}>Search</button>
                </div> */}
              </div>

              <div className="x_content">

                <DataTableComponent
                  rowData={competitorData}
                  columnDefs={competitorColumn}
                  loading={loading}
                  topMarket={true}
                />
                <TotalRow
                  rowData={total}
                  columnDefs={competitorColumn}
                  loading={loading}
                />

                <div className='row' style={{ marginTop: '10px' }}>
                  <div className='col-md-6 col-sm-6 col-xs-12'>
                    {loading2 ? '' : topMarketData.length > 0 ? <h2 className='table-name'>{tableTitle1}</h2> : <h2 />}
                    <DataTableComponent
                      rowData={topMarketData}
                      columnDefs={topMarketColumn}
                      onCellClicked={(cellData) => this.topMarketRowClick(cellData)}
                      topMarket={true}
                      loading={loading2}
                      rowClassRules={this.state.topMarketRowHighlighted}
                    />
                    <TotalRow
                      rowData={topMarketTotal}
                      columnDefs={topMarketColumn}
                      loading={loading2}
                    />
                  </div>
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
                </div>

              </div>
            </div>
          </div>
        </div>
      </div >

    );
  }
}

export default CompetitorAnalysis;