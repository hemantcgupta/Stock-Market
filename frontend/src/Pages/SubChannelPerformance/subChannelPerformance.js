import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi';
import DataTableComponent from '../../Component/DataTableComponent';
import DownloadCustomHeaderGroup from './DownloadCustomHeaderGroup'
import TotalRow from '../../Component/TotalRow';
import RegionsDropDown from '../../Component/RegionsDropDown';
import './SubChannelPerformance.scss';
import TopMenuBar from '../../Component/TopMenuBar';
import URLSearchParams from '../../Constants/validator';
import _ from 'lodash';

const apiServices = new APIServices();

class SubChannelPerformance extends Component {

  constructor(props) {
    super(props);
    this.state = {
      startDate: null,
      endDate: null,
      startDate: null,
      getCabinValue: null,
      regionSelected: null,
      countrySelected: null,
      citySelected: null,
      channel:null,
      subChannelColumn: [],
      subChannelData: [],
      totalData: [],
      loading: false,
      getAgent:'Null',
      title: 'Sub Channel Performance of',
    };
    this.sendEvent('1', 'viewed Sub Channel Performance Page', 'subChannelPerformance', 'Sub Channel Performance');
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
    let getAgent = URLSearchParams.getParameterByName('subchannel', window.location.href)
    this.setState({
      channel:URLSearchParams.getParameterByName('channel', window.location.href),
      getAgent: getAgent === null ? this.state.getAgent : getAgent
    }) 
  }

  getFilterValues = ($event) => {
    this.setState({
      regionSelected: $event.regionSelected === 'All' ? 'Null' : $event.regionSelected,
      countrySelected: $event.countrySelected,
      citySelected: $event.citySelected,
      startDate: $event.startDate,
      endDate: $event.endDate,
      getCabinValue: $event.getCabinValue,
    }, () => this.getSubChannelPerformanceData())
  }

  getSubChannelPerformanceData = () => {
    var self = this;
    let { endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue,channel,getAgent} = this.state;
    self.setState({ loading: true })

    let title = this.state.getAgent === 'Null' ? `Sub Channel Performance of ${channel}` : `Top OD's for ${channel} & ${getAgent}`

    apiServices.getSubChannelPerformance(endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, channel, getAgent).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var columnName = result[0].columnName;
        var rowData = result[0].rowData;
        var totalData = result[0].totalData;
        self.setState({
          subChannelData: rowData, totalData: totalData, subChannelColumn: columnName,
          title: title
        })
      }
    });
  }

  channelRowClick = (params) => {
    var rowData = params.api.getSelectedRows();
    var getAgent = rowData[0].Agents;
    var column = params.colDef.field;
    let channel = this.state.channel;
    if(column === 'Agents') {
      this.props.history.push(`/subChannelODPerformance?channel=${channel}&subchannel=${encodeURIComponent(getAgent)}`)
    }
    // else if (column === 'CommonOD') {
    //   // this.props.history.push(`/subSubChannelPerformance?${channel}`)
    //   var self = this;
    //   let { endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, title,channel } = this.state;
    //   self.setState({ loading: true})

    //   apiServices.getSubChannelPerformance(endDate, startDate, regionSelected, countrySelected, citySelected, getCabinValue, channel, getAgent).then(function (result) {
    //     self.setState({ loading: false })
    //     if (result) {
    //       var columnName = result[0].columnName;
    //       var rowData = result[0].rowData;
    //       var totalData = result[0].totalData;
    //       self.setState({
    //         subChannelData: rowData, totalData: totalData, subChannelColumn: columnName,
    //         title: `Top OD's for ${channel} & ${getAgent}`,
    //         getAgent:getAgent,
    //       })
    //     }
    //   });
    // }
  }


  render() {
    const { loading, title } = this.state;

    return (
      <div className='sub-channel-performance'>
        <TopMenuBar {...this.props} />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 sub-channel-performance-main">
            <div className="navdesign">
              <div className="col-md-12 col-sm-12 col-xs-12 top-heading">
                <section>
                  <h2>{title}</h2>
                </section>
              </div>

              <RegionsDropDown
                getFilterValues={this.getFilterValues}
                hideCabin={true}
                {...this.props} />

              <div className="x_content" style={{ marginTop: '15px', padding: '10px' }}>
                <DataTableComponent
                  rowData={this.state.subChannelData}
                  columnDefs={this.state.subChannelColumn}
                  onCellClicked={(cellData) => this.channelRowClick(cellData)}
                  frameworkComponents={{ customHeaderGroupComponent: DownloadCustomHeaderGroup }}
                  loading={loading}
                  channel={true}
                />
                <TotalRow
                  rowData={this.state.totalData}
                  columnDefs={this.state.subChannelColumn}
                  frameworkComponents={{ customHeaderGroupComponent: DownloadCustomHeaderGroup }}
                  loading={loading}
                  reducingPadding={true}
                />
              </div>
            </div>

          </div>
        </div>

      </div >

    );
  }
}

export default SubChannelPerformance;