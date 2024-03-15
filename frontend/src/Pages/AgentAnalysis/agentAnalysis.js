import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi';
import api from '../../API/api';
import DataTableComponent from '../../Component/DataTableComponent';
import TotalRow from '../../Component/TotalRow';
import RegionsDropDown from '../../Component/RegionsDropDown';
import color from '../../Constants/color'
import $ from 'jquery';
import '../../App';
import './AgentAnalysis.scss';
import TopMenuBar from '../../Component/TopMenuBar';
import AutoSelectPagination from '../../Component/AutoSelectPagination';
import _ from 'lodash';

const apiServices = new APIServices();

const options = [];
for (let i = 0; i < 50; ++i) {
  options.push({
    value: i + 1,
    label: `Option ${i + 1}`
  });
}

class AgentAnalysis extends Component {
  constructor(props) {
    super(props);
    this.state = {
      startDate: null,
      endDate: null,
      regionSelected: 'Null',
      countrySelected: 'Null',
      citySelected: 'Null',
      topMarketNum: ['5', '10', '15', '20'],
      selectedTopMarketNum: '5',
      agentColumn: [],
      agentData: [],
      topMarketColumn: [],
      topMarketData: [],
      topMarketTotal: [],
      odName: '',
      modelRegionDatas: [],
      modelregioncolumn: [],
      tableDatas: true,
      tableTitle1: '',
      // monthTableTitle:'',
      top5ODTitle: 'TOP 5 ODs',
      tabLevel: 'Null',
      posFlowData: 'Null',
      posAgentFlowDatas: [],
      posAgentFlowcolumn: [],
      getCabinValue: [],
      agents: [],
      agentValue: [],
      agentName: '',
      pageNumber: '',
      totalRecords: '',
      paginationSize: '',
      loading: false,
      loading2: false
    };
    this.sendEvent('1', 'viewed Agent Analysis Page', 'agentAnalysis', 'Agent Analysis');

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
    var agentName = window.localStorage.getItem('Agent')
    this.setState({
      agentValue: [{ value: '1', label: agentName }],
      agentName: agentName,
      tableTitle1: `Top Markets for ${agentName}`,
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
    }, () => {
      this.getTopMarketData(); this.getAgentAnalysisReport();
    })
  }

  getAgentAnalysisReport = () => {
    var self = this;
    let { endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, agentName } = this.state;
    self.setState({ loading: true })

    apiServices.getAgentAnalysis(endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, agentName).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        self.setState({
          agentData: rowData,
          agentColumn: columnName,
          tableTitle1: `Top Markets for ${agentName}`
        })
      }
    });
  }

  getTopMarketData = () => {
    var self = this;
    let { endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, selectedTopMarketNum, agentName } = this.state;
    self.setState({ loading2: true })

    apiServices.getTopMarketsForAgent(endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, selectedTopMarketNum, agentName).then(function (result) {
      self.setState({ loading2: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        self.setState({
          topMarketData: rowData,
          topMarketTotal: totalData,
          topMarketColumn: columnName
        })
      }
    });
  }

  callAgents = (value) => {
    this.sendEvent('2', 'clicked on Agents Dropdown', 'agentAnalysis', 'Agent Analysis');
    this.setState({ agentValue: value, agentName: value.label }, () => {
      this.getTopMarketData(); this.getAgentAnalysisReport();
    })
  }

  callTopMarket = (e) => {
    this.sendEvent('2', 'clicked on Top Markets Dropdown', 'agentAnalysis', 'Agent Analysis');
    this.setState({ selectedTopMarketNum: e.target.value }, () => this.getTopMarketData())
  }

  search = () => {
    this.sendEvent('2', 'clicked on Search button', 'agentAnalysis', 'Agent Analysis');
    this.getAgentAnalysisReport();
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

  getAgents = (agentName, pageNumber) => {
    return new Promise((resolve, reject) => {
      const params = {};
      if (agentName) {
        params['agent_name'] = agentName
      }
      if (pageNumber) {
        params['page_number'] = pageNumber
      }
      api.get(`getagent?${this.serialize(params)}`)
        .then((response) => {
          if (response) {
            if (response.data.response) {
              let data = response.data.response;
              resolve({
                agents: data.agents,
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
    const optionsPerPage = 10;

    const result = await this.getAgents(search, page);
    if (result === null) {
      return
    }
    const {
      agents,
      totalRecords,
      totalPages
    } = result;

    if (agents.length > 0) {
      agents.map(agent => {
        if (agent.label) {
          const s = agent.label.trim().toLowerCase();
          agent.label = s.charAt(0).toUpperCase() + s.slice(1);
        }
        return agent
      });
      return {
        options: agents,
        hasMore: totalPages > page + 1
      };
    }
  };

  render() {
    const { loading, loading2, agentData, agentColumn, topMarketData, topMarketTotal, topMarketColumn, startDate, tableTitle1, agentName, cabinOption, topMarketNum } = this.state;

    const topMarketOptionItems = topMarketNum.map((num) =>
      <option value={num}>{num}</option>
    );

    return (
      <div className='agent-analysis'>
        <TopMenuBar {...this.props} />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 agent-analysis-main">
            <div className="navdesign">
              <div className="col-md-12 col-sm-12 col-xs-12 top-heading">
                <section>
                  <h2>Agent Analysis</h2>
                </section>
                <div style={{ display: 'flex' }}>
                  <div className="" >
                    <div className="form-group" style={{ maxWidth: '300px' }}>
                      <h4>Agents :</h4>
                      <AutoSelectPagination
                        value={this.state.agentValue}
                        onChange={(value) => this.callAgents(value)}
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
                  rowData={agentData}
                  columnDefs={agentColumn}
                  autoHeight={'autoHeight'}
                  loading={loading}
                />
              </div>
            </div>

            <div className="x_content">
              {loading2 ? '' : topMarketData.length > 0 ? <h2 className='table-name'>{tableTitle1}</h2> : <h2 />}
              <DataTableComponent
                rowData={topMarketData}
                columnDefs={topMarketColumn}
                topMarket={true}
                loading={loading2}
              />
              <TotalRow
                rowData={topMarketTotal}
                columnDefs={topMarketColumn}
                loading={loading2}
              />
            </div>
          </div>
        </div>
      </div >

    );
  }
}

export default AgentAnalysis;